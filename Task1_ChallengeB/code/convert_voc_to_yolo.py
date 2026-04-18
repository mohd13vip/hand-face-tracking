import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

VOC_ROOT = Path("VOC2012_train_val/VOC2012_train_val")
OUT_ROOT = Path("VOC2012_YOLO")

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

def voc_box_to_yolo(xmin, ymin, xmax, ymax, w, h):
    xmin = max(0.0, xmin - 1.0)
    ymin = max(0.0, ymin - 1.0)
    xmax = max(0.0, xmax - 1.0)
    ymax = max(0.0, ymax - 1.0)

    bw = xmax - xmin
    bh = ymax - ymin
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0

    return (
        clamp(cx / w),
        clamp(cy / h),
        clamp(bw / w),
        clamp(bh / h),
    )

def ensure_dirs():
    for split in ["train", "val"]:
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

def load_split(split):
    p = VOC_ROOT / "ImageSets" / "Main" / f"{split}.txt"
    return [x.strip() for x in p.read_text().splitlines() if x.strip()]

def convert_split(split):
    ids = load_split(split)
    print(f"Converting {split}: {len(ids)} images")

    for img_id in tqdm(ids):
        xml_path = VOC_ROOT / "Annotations" / f"{img_id}.xml"
        img_path = VOC_ROOT / "JPEGImages" / f"{img_id}.jpg"

        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        w = float(size.find("width").text)
        h = float(size.find("height").text)

        yolo_lines = []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            if name not in CLASSES:
                continue

            difficult = obj.find("difficult")
            if difficult is not None and difficult.text.strip() == "1":
                continue

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            cx, cy, bw, bh = voc_box_to_yolo(xmin, ymin, xmax, ymax, w, h)
            cls_id = CLASSES.index(name)
            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        if yolo_lines:
            (OUT_ROOT / "labels" / split / f"{img_id}.txt").write_text("".join(yolo_lines))
            shutil.copy2(img_path, OUT_ROOT / "images" / split / f"{img_id}.jpg")

def write_yaml():
    yaml_text = f"""path: {OUT_ROOT.resolve()}
train: images/train
val: images/val

nc: 20
names: {CLASSES}
"""
    (OUT_ROOT / "data.yaml").write_text(yaml_text)

def main():
    ensure_dirs()
    convert_split("train")
    convert_split("val")
    write_yaml()
    print("\n✅ Done!")
    print(f"Dataset folder: {OUT_ROOT}")
    print(f"YAML: {OUT_ROOT / 'data.yaml'}")

if __name__ == "__main__":
    main()

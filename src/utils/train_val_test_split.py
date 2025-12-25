import os
import random
import shutil

# =========================
# CONFIG
# =========================
IMAGE_DIR = "data/images"
XML_DIR = "data/annotations/xml"

OUTPUT_BASE = "data/splits"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

RANDOM_SEED = 42

# =========================
# UTILS
# =========================
def make_dirs():
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_BASE, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_BASE, split, "annotations"), exist_ok=True)

def get_image_xml_pairs():
    pairs = []
    for file in os.listdir(XML_DIR):
        if file.endswith(".xml"):
            xml_path = os.path.join(XML_DIR, file)
            image_name = file.replace(".xml", ".png")

            image_path = os.path.join(IMAGE_DIR, image_name)
            if os.path.exists(image_path):
                pairs.append((image_path, xml_path))
    return pairs

def split_data(pairs):
    random.seed(RANDOM_SEED)
    random.shuffle(pairs)

    total = len(pairs)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train = pairs[:train_end]
    val = pairs[train_end:val_end]
    test = pairs[val_end:]

    return train, val, test

def copy_files(pairs, split_name):
    for img_path, xml_path in pairs:
        shutil.copy(img_path, os.path.join(OUTPUT_BASE, split_name, "images"))
        shutil.copy(xml_path, os.path.join(OUTPUT_BASE, split_name, "annotations"))

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    make_dirs()

    pairs = get_image_xml_pairs()
    print(f"Total valid image-XML pairs: {len(pairs)}")

    train, val, test = split_data(pairs)

    copy_files(train, "train")
    copy_files(val, "val")
    copy_files(test, "test")

    print(f"Train: {len(train)}")
    print(f"Validation: {len(val)}")
    print(f"Test: {len(test)}")

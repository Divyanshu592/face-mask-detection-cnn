import os
import tensorflow as tf
import xml.etree.ElementTree as ET
import cv2

CLASSES = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def parse_xml(xml_path, image_dir):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text.strip()
    image_path = os.path.join(image_dir, filename)

    if not os.path.exists(image_path):
        return None

    image = cv2.imread(image_path)
    if image is None:
        return None

    height, width, _ = image.shape

    xmins, ymins, xmaxs, ymaxs, labels = [], [], [], [], []

    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip()
        if class_name not in CLASSES:
            continue

        bndbox = obj.find("bndbox")
        xmins.append(float(bndbox.find("xmin").text) / width)
        ymins.append(float(bndbox.find("ymin").text) / height)
        xmaxs.append(float(bndbox.find("xmax").text) / width)
        ymaxs.append(float(bndbox.find("ymax").text) / height)
        labels.append(CLASSES[class_name])

    if len(labels) == 0:
        return None

    return image_path, xmins, ymins, xmaxs, ymaxs, labels

def create_example(image_path, xmins, ymins, xmaxs, ymaxs, labels):
    with tf.io.gfile.GFile(image_path, "rb") as f:
        encoded_image = f.read()

    feature = {
        "image/encoded": _bytes_feature(encoded_image),
        "image/filename": _bytes_feature(os.path.basename(image_path).encode()),
        "image/object/bbox/xmin": _float_list_feature(xmins),
        "image/object/bbox/ymin": _float_list_feature(ymins),
        "image/object/bbox/xmax": _float_list_feature(xmaxs),
        "image/object/bbox/ymax": _float_list_feature(ymaxs),
        "image/object/class/label": _int64_list_feature(labels),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

def convert_split(split):
    image_dir = f"data/splits/{split}/images"
    xml_dir = f"data/splits/{split}/annotations"
    output_path = f"data/annotations/tfrecords/{split}.tfrecord"

    os.makedirs("data/annotations/tfrecords", exist_ok=True)

    writer = tf.io.TFRecordWriter(output_path)
    count = 0

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith(".xml"):
            continue

        parsed = parse_xml(os.path.join(xml_dir, xml_file), image_dir)
        if parsed is None:
            continue

        example = create_example(*parsed)
        writer.write(example.SerializeToString())
        count += 1

    writer.close()
    print(f"{split}: wrote {count} samples")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        convert_split(split)

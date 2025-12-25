import os
import xml.etree.ElementTree as ET

CLASSES = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}

def parse_voc_xml(xml_path):
    """
    Parse a Pascal VOC XML file that may contain multiple objects (faces).

    Returns:
        image_info (dict):
            {
              "filename": str,
              "width": int,
              "height": int,
              "objects": [
                  {
                      "bbox": [xmin, ymin, xmax, ymax],
                      "label": int
                  },
                  ...
              ]
            }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip()

        if class_name not in CLASSES:
            continue  # ignore unknown classes

        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        objects.append({
            "bbox": [xmin, ymin, xmax, ymax],
            "label": CLASSES[class_name]
        })

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects
    }


def parse_annotation_folder(xml_dir):
    """
    Parse all XML files in a directory.

    Returns:
        List of parsed image annotations
    """
    annotations = []

    for file in os.listdir(xml_dir):
        if file.endswith(".xml"):
            xml_path = os.path.join(xml_dir, file)
            parsed = parse_voc_xml(xml_path)

            if len(parsed["objects"]) > 0:
                annotations.append(parsed)

    return annotations


if __name__ == "__main__":
    # Quick sanity check
    xml_directory = "data/annotations/xml"
    parsed_data = parse_annotation_folder(xml_directory)
    print(f"Parsed {len(parsed_data)} annotation files")

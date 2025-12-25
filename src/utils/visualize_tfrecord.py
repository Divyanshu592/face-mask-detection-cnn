import tensorflow as tf
import matplotlib.pyplot as plt

TFRECORD_PATH = "data/annotations/tfrecords/train.tfrecord"
IMAGE_SIZE = 224
NUM_IMAGES = 5

def parse_tfrecord(example):
    feature_description = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
    }

    example = tf.io.parse_single_example(example, feature_description)

    image = tf.image.decode_png(example["image/encoded"], channels=3)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32) / 255.0

    xmins = tf.sparse.to_dense(example["image/object/bbox/xmin"])
    ymins = tf.sparse.to_dense(example["image/object/bbox/ymin"])
    xmaxs = tf.sparse.to_dense(example["image/object/bbox/xmax"])
    ymaxs = tf.sparse.to_dense(example["image/object/bbox/ymax"])

    boxes = tf.stack([ymins, xmins, ymaxs, xmaxs], axis=1)
    return image, boxes

dataset = tf.data.TFRecordDataset(TFRECORD_PATH)
dataset = dataset.map(parse_tfrecord)

for image, boxes in dataset.take(NUM_IMAGES):
    boxes = tf.expand_dims(boxes, axis=0)
    image = tf.expand_dims(image, axis=0)

    colors = tf.constant([[1.0, 0.0, 0.0]])  # red boxes
    image_with_boxes = tf.image.draw_bounding_boxes(image, boxes, colors)

    plt.figure(figsize=(5, 5))
    plt.imshow(image_with_boxes[0])
    plt.axis("off")
    plt.show()

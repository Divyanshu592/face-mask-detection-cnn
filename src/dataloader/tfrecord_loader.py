import tensorflow as tf

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_CLASSES = 3

def parse_tfrecord(example):
    feature_description = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
        "image/object/class/label": tf.io.VarLenFeature(tf.int64),
    }

    example = tf.io.parse_single_example(example, feature_description)

    image = tf.image.decode_png(example["image/encoded"], channels=3)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32) / 255.0

    xmins = tf.sparse.to_dense(example["image/object/bbox/xmin"])
    ymins = tf.sparse.to_dense(example["image/object/bbox/ymin"])
    xmaxs = tf.sparse.to_dense(example["image/object/bbox/xmax"])
    ymaxs = tf.sparse.to_dense(example["image/object/bbox/ymax"])
    labels = tf.sparse.to_dense(example["image/object/class/label"])

    boxes = tf.stack([ymins, xmins, ymaxs, xmaxs], axis=1)
    labels = tf.one_hot(labels[0], NUM_CLASSES)  # take first face


    return image, {"boxes": boxes[0], "labels": labels}


def augment(image, targets):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image, targets

def load_dataset(tfrecord_path, training=True):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(256)
    
    dataset = dataset.batch(BATCH_SIZE)

    return dataset.prefetch(tf.data.AUTOTUNE)

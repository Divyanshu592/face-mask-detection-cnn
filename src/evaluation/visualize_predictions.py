import tensorflow as tf
import matplotlib.pyplot as plt
import os

MODEL_PATH = "models/saved_model/mask_detector"
TFRECORD_PATH = "data/annotations/tfrecords/test.tfrecord"
OUTPUT_DIR = "reports/visualizations"

IMAGE_SIZE = 224
NUM_IMAGES = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def main():
    model = tf.keras.models.load_model(MODEL_PATH)

    dataset = tf.data.TFRecordDataset(TFRECORD_PATH)
    dataset = dataset.map(parse_tfrecord)

    for idx, (image, true_boxes) in enumerate(dataset.take(NUM_IMAGES)):
        image_batch = tf.expand_dims(image, axis=0)

        preds = model.predict(image_batch, verbose=0)
        pred_box = preds["boxes"][0]

        pred_box = tf.reshape(pred_box, (1, 1, 4))
        image_boxed = tf.image.draw_bounding_boxes(
            tf.expand_dims(image, axis=0),
            pred_box,
            colors=[[1.0, 0.0, 0.0]]
        )

        plt.figure(figsize=(5, 5))
        plt.imshow(image_boxed[0])
        plt.axis("off")

        out_path = os.path.join(OUTPUT_DIR, f"prediction_{idx}.png")
        plt.savefig(out_path)
        plt.close()

        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()

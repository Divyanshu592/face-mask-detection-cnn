import tensorflow as tf
from tensorflow.keras import layers, models

IMAGE_SIZE = 224
NUM_CLASSES = 3

def build_model():
    # Backbone
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = False

    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # Bounding box head
    bbox_dense = layers.Dense(256, activation="relu")(x)
    bbox_output = layers.Dense(4, activation="sigmoid", name="boxes")(bbox_dense)

    # Classification head
    cls_dense = layers.Dense(256, activation="relu")(x)
    cls_output = layers.Dense(NUM_CLASSES, activation="softmax", name="labels")(cls_dense)

    model = models.Model(inputs=inputs, outputs={
        "boxes": bbox_output,
        "labels": cls_output
    })

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            "boxes": "mse",
            "labels": "categorical_crossentropy"
        }
    )

    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()

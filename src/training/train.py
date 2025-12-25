import tensorflow as tf
from src.dataloader.tfrecord_loader import load_dataset
from src.model.detector_model import build_model

TRAIN_TFRECORD = "data/annotations/tfrecords/train.tfrecord"
VAL_TFRECORD = "data/annotations/tfrecords/val.tfrecord"

EPOCHS = 10

def main():
    train_ds = load_dataset(TRAIN_TFRECORD, training=True)
    val_ds = load_dataset(VAL_TFRECORD, training=False)

    model = build_model()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/checkpoints/best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save("models/saved_model/mask_detector")

if __name__ == "__main__":
    main()

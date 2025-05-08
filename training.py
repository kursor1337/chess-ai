from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam # type: ignore
import tensorflow as tf
from dataset import chess_train_data_generator
from dataset import chess_validation_data_generator
from model import build_model

train_dataset = tf.data.Dataset.from_generator(
    chess_train_data_generator,
    output_types=(tf.float32, tf.int16),
    output_shapes=((8, 8, 12), ())
)

validation_dataset = tf.data.Dataset.from_generator(
    chess_validation_data_generator,
    output_types=(tf.float32, tf.int16),
    output_shapes=((8, 8, 12), ())
)

train_dataset = train_dataset.repeat(3).batch(32)
validation_dataset = validation_dataset.batch(32)

model = build_model()

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(train_dataset, validation_data=validation_dataset)
model.save("models/TF_50EPOCHS.keras")


import tensorflow as tf
from tensorflow.keras import layers, models


def residual_block(x, filters, downsample=False):
    shortcut = x

    strides = 2 if downsample else 1

    # First conv layer
    x = layers.Conv2D(filters, 3, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second conv layer
    x = layers.Conv2D(filters, 3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Downsample shortcut if needed
    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add skip connection
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)

    return x


def build_resnet(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Residual blocks
    for _ in range(4):
        x = residual_block(x, 64)

    # Stage 2: 4 blocks, 128 filters, downsample at start
    x = residual_block(x, 128, downsample=True)
    for _ in range(3):
        x = residual_block(x, 128)

    # Stage 3: 4 blocks, 256 filters, downsample at start
    x = residual_block(x, 256, downsample=True)
    for _ in range(3):
        x = residual_block(x, 256)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, x)
    return model


def build_model():
    return build_resnet(input_shape=(8, 8, 12), num_classes=64 * 64)


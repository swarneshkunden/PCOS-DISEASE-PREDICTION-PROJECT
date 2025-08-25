# cbam.py

import tensorflow as tf
from tensorflow.keras import layers, models

class CBAM(layers.Layer):
    def __init__(self, channels, reduction_ratio=16, **kwargs):
        super(CBAM, self).__init__(**kwargs)  # Pass kwargs to the parent class constructor
        self.channel_attention = self._build_channel_attention(channels, reduction_ratio)
        self.spatial_attention = self._build_spatial_attention()

    def _build_channel_attention(self, channels, reduction_ratio):
        return models.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Reshape((1, 1, channels)),
            layers.Dense(channels // reduction_ratio, activation='relu'),
            layers.Dense(channels, activation='sigmoid')
        ])

    def _build_spatial_attention(self):
        return models.Sequential([
            layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')
        ])

    def call(self, inputs):
        # Apply Channel Attention
        channel_attention = self.channel_attention(inputs)
        x = layers.Multiply()([inputs, channel_attention])

        # Apply Spatial Attention
        spatial_attention = self.spatial_attention(x)
        x = layers.Multiply()([x, spatial_attention])

        return x

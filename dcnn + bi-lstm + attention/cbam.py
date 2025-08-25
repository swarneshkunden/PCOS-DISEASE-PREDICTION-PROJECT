from tensorflow.keras import layers, models

class CBAM(layers.Layer):
    def __init__(self, channels, reduction_ratio=16, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.channel_attention = self._build_channel_attention()
        self.spatial_attention = self._build_spatial_attention()

    def _build_channel_attention(self):
        # Channel attention mechanism using global average pooling
        return models.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Reshape((1, 1, self.channels)),  # Reshape to (1, 1, channels)
            layers.Dense(self.channels // self.reduction_ratio, activation='relu'),
            layers.Dense(self.channels, activation='sigmoid')
        ])

    def _build_spatial_attention(self):
        # Spatial attention using a convolutional layer with a kernel size of (7, 7)
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

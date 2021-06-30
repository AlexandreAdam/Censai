import tensorflow as tf


class DownsamplingLayer(tf.keras.Model):
    def __init__(
            self,
            method: str,
            layers: int,
            conv_layers: int,
            strides: int,
            filters: int,
            output_filters: int,
            kernel_size: 5):
        super(DownsamplingLayer, self).__init__()
        method = "conv"  # Overwrite method since we only support conv for nor this layer for now
        self.output_layer = tf.keras.layers.Conv2D(filters=output_filters, kernel_size=kernel_size, padding="SAME")
        self._layers = []
        for i in range(layers):
            for j in range(conv_layers):
                self._layers.append(
                    tf.keras.layers.Conv2D(
                        filters=filters,
                        kernel_size=kernel_size,
                        padding="SAME"
                    )
                )
            if method == "conv":
                self._layers.append(
                    tf.keras.layers.Conv2D(
                        filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding="SAME",
                    )
                )
            else:
                raise NotImplemented(f"{method} not in [conv]")

    def call(self, kappa):
        for layer in self._layers:
            kappa = layer(kappa)
        kappa = self.output_layer(kappa)
        return kappa
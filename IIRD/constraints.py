import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class Positive(tf.keras.constraints.Constraint):
    """Makes sure non negative by clipping -ve values to 0"""
    def __call__(self, w):
        return tf.maximum(w, 0.0)  # Clip negative values to 0
        # return tf.nn.relu(w)
        return w

    def get_config(self):
        return {}  # No parameters to serialize

# class No(Constraint):
#     def __call__(self, w):
#         # return tf.maximum(w, 0.0)  # Clip negative values to 0
#         # return tf.nn.relu(w)
#         return w
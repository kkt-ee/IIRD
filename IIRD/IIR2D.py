## OK general IIR 2D layer
import tensorflow as tf
from IIRD.constraints import Positive

# @tf.keras.utils.register_keras_serializable()
# class Positive(tf.keras.constraints.Constraint):
#     """Makes sure non negative by clipping -ve values to 0"""
#     def __call__(self, w):
#         return tf.maximum(w, 0.0)  # Clip negative values to 0
#         # return tf.nn.relu(w)
#         return w

#     def get_config(self):
#         return {}  # No parameters to serialize

# class No(Constraint):
#     def __call__(self, w):
#         # return tf.maximum(w, 0.0)  # Clip negative values to 0
#         # return tf.nn.relu(w)
#         return w

@tf.keras.utils.register_keras_serializable()
class IIR2D(tf.keras.layers.Layer):
    """ IIR 2D layer
    
    IIRD: Fast trainable D-dimensional IIR filter layers in TensorFlow.
    Copyright 2025 Kishore Kumar Tarafdar.
    Licensed under the Apache License, Version 2.0. See LICENSE for details.
    """
    def __init__(self, Delays:int, filters:int, tolerance=1e-6, max_steps=5000, local_lr=0.001, **kwargs):
        super().__init__(**kwargs)
        self.Delays = Delays
        self.filters = filters ## number of out channels
        self.tolerance = tolerance
        self.max_steps = max_steps
        self.local_lr = local_lr
        # Define optimizer (can be class attribute if needed)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.local_lr)

    def build(self, input_shape):
        self.N = input_shape[1]
        self.channels = input_shape[-1]

        # Create grid of shifts: shape ((D+1)**2, 2)
        shifts = tf.range(self.Delays + 1)
        shift_grid = tf.stack(tf.meshgrid(shifts, shifts, indexing="ij"), axis=-1)  # (D+1, D+1, 2)
        self.shift_grid = tf.reshape(shift_grid, [-1, 2])  # (num_shifts, 2)
        
        ## IIR feed forward coeffs
        self.b = self.add_weight(
            shape=(self.channels, int((self.Delays + 1)**2), self.filters), 
            initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=0.1), 
            trainable=True, 
            name="b_coeffs",
            constraint=Positive() ## no constraint
        )
        ## IIR feed back coeffs
        self.a = self.add_weight(
            shape=(self.channels, int((self.Delays + 1)**2-1), self.filters), 
            initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=0.1), 
            trainable=True, 
            name="a_coeffs",
            constraint=Positive() ## constraint >0 for STABLE FILTER
        )
        super().build(input_shape)
        # ## Setting a0=1 in feedback coeffs
        # a0s = tf.ones((self.channels, 1, self.filters))
        # self.A = tf.concat([a0s, self.a], axis=-2)  # shape: (channels, Delays+1)

        # self.output_shape = (input_shape[0],input_shape[1], input_shape[2], self.filters) 

       
    # @tf.function(jit_compile=False)  # Disable XLA for debugging
    def call(self, x):
        ## Setting a0=1 in feedback coeffs
        a0s = tf.ones((self.channels, 1, self.filters))
        A = tf.concat([a0s, self.a], axis=-2)  # shape: (channels, Delays+1)
        
        ## initializing IIR filter or layer output y
        # y = tf.identity(x)  # start from input
        # y = tf.zeros_like(x)
        y = tf.random.uniform(
            shape=(tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3],  self.filters),
            minval=0,
            maxval=1
            )

        ## Creating shifted tensor: [x[n], x[n-1], ... , x[n-D]] 
        Xshifted = self.__circularshift_x2d(x)  # Precompute once
        # self.Xshifted = Xshifted

        ## Local optimization loss
        def compute_residual(y_var):
            """Compute Residual |AX-BY| for local optimization"""
            Yshifted = self.__circularshift_y2d(y_var)
            # print('+',Xshifted.shape, self.b.shape)
            # print('+',Yshifted.shape, self.a.shape)
            BX = tf.einsum('bnmcdo,cdo->bnmo', Xshifted, self.b)
            AY = tf.einsum('bnmcdo,cdo->bnmo', Yshifted, A)
            
            residual = BX - AY
            # loss = tf.reduce_sum(residual) # unstable filters
            # loss = tf.reduce_sum(tf.abs(residual)) #better
            loss = tf.reduce_sum(tf.square(residual)) #better
            return loss, residual

        # Optimization loop------
        i = tf.constant(0)
        loss_init, _ = compute_residual(y)

        # i = tf.constant(0, dtype=tf.int32)
        # loss_init = tf.constant(float('inf'), dtype=x.dtype)

        def cond(i, y_var, loss):
            # print(loss.numpy(), self.tolerance, (loss > self.tolerance).numpy())
            return tf.logical_and(loss > self.tolerance, i < self.max_steps)
        
        def body(i, y_var, loss):
            with tf.GradientTape() as tape:
                # print(f'{i}', end=', ')
                tape.watch(y_var)
                loss, _ = compute_residual(y_var)
            grads = tape.gradient(loss, [y_var] + self.trainable_variables)
            dy = grads[0]
            # dvars = grads[1:]

            # Update y manually
            y_var = y_var - self.local_lr * dy
            return i + 1, y_var, loss

        i, y_opt, loss_final = tf.while_loop(
            cond, 
            body,
            loop_vars=[i, y, loss_init],
            # maximum_iterations=self.max_steps,  # Critical for XLA
            # shape_invariants=[
            #     i.shape,
            #     tf.TensorShape([None, self.N, self.channels]),  # Batch dimension can vary
            #     loss_init.shape
            # ]
            # maximum_iterations=self.max_steps
            # maximum_iterations=None
        )
        ## END local optimization----------               
        y = tf.einsum('bnmco->bnmo', y_opt)
        # print('y shape' ,y.shape)
        # y = tf.expand_dims(, axis=-1)
        return y

    def __circularshift_x2d(self, x):
        """
        x: Tensor of shape (batch, N, N, channels)
        delays D: int, number of delays in each dimension
        out_channels: int, number of output channels (identical copies)
        
        Returns:
            Tensor of shape (batch, N, N, channels, (D+1)**2, out_channels)
        """
        # batch, N, _, channels = tf.unstack(tf.shape(x))
        batch, N, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[3]
        # out_channels = self.filters
        # num_shifts = tf.shape(self.shift_grid)[0]  # (D+1)^2

        # # Create grid of shifts: shape ((D+1)**2, 2)
        # shifts = tf.range(self.Delays + 1)
        # shift_grid = tf.stack(tf.meshgrid(shifts, shifts, indexing="ij"), axis=-1)  # (D+1, D+1, 2)
        # self.shift_grid = tf.reshape(shift_grid, [-1, 2])  # (num_shifts, 2)

        # Broadcast input x to (num_shifts, batch, N, N, channels)
        x_broadcasted = tf.expand_dims(x, axis=0)
        x_broadcasted = tf.repeat(x_broadcasted, repeats=tf.shape(self.shift_grid)[0], axis=0)

        # Roll each slice using vectorized shift
        def single_roll(args):
            x_i, shift = args
            return tf.roll(x_i, shift=[-shift[0], -shift[1]], axis=[1, 2])

        rolled = tf.map_fn(
            single_roll, 
            (x_broadcasted, self.shift_grid), 
            fn_output_signature=tf.TensorSpec(shape=(None, None, None, None), dtype=x.dtype))

        # rolled shape: (num_shifts, batch, N, N, channels) → transpose to (batch, N, N, channels, num_shifts)
        rolled = tf.transpose(rolled, perm=[1, 2, 3, 4, 0])

        # Expand and tile to (batch, N, N, channels, num_shifts, out_channels)
        rolled = tf.expand_dims(rolled, axis=-1)
        rolled = tf.tile(rolled, [1, 1, 1, 1, 1, self.filters])

        return rolled

    def __circularshift_y2d(self, y):
        """
        y: Tensor of shape (batch, N, N, channels, out_channels)
        delays: int, number of delays in each dimension

        Returns:
            Tensor of shape (batch, N, N, channels, (D+1)**2, out_channels)
        """
        # batch, N, _, channels,  = tf.unstack(tf.shape(y))
        # batch, N, _, channels, out_channels = tf.shape(y)
        batch, N, channels, out_channels = tf.shape(y)[0], tf.shape(y)[1], tf.shape(y)[-2], tf.shape(y)[-1]

        # Create grid of shifts: shape ((D+1)**2, 2)
        # shifts = tf.range(delays + 1)
        # shift_grid = tf.stack(tf.meshgrid(shifts, shifts, indexing="ij"), axis=-1)  # (D+1, D+1, 2)
        # shift_grid = tf.reshape(shift_grid, [-1, 2])  # (num_shifts, 2)

        # Broadcast y to shape (num_shifts, batch, N, N, channels, out_channels)
        y_broadcasted = tf.expand_dims(y, axis=0)
        y_broadcasted = tf.repeat(y_broadcasted, repeats=tf.shape(self.shift_grid)[0], axis=0)

        # Roll each slice using vectorized shift
        def single_roll(args):
            y_i, shift = args
            return tf.roll(y_i, shift=[-shift[0], -shift[1]], axis=[1, 2])

        rolled = tf.map_fn(
            single_roll,
            (y_broadcasted, self.shift_grid),
            fn_output_signature=tf.TensorSpec(shape=(None, None, None, None, None), dtype=y.dtype)
        )

        # rolled shape: (num_shifts, batch, N, N, channels, out_channels)
        # → transpose to (batch, N, N, channels, num_shifts, out_channels)
        rolled = tf.transpose(rolled, perm=[1, 2, 3, 4, 0, 5])

        return rolled

    def get_config(self):
        config = super().get_config()
        config.update({
            'Delays': self.Delays,
            'filters': self.filters,
            'tolerance': self.tolerance,
            'max_steps': self.max_steps,
            'local_lr': self.local_lr,
        })
        return config



if __name__=='__main__':
    # Example usage:
    batch_size = 10
    N = 128
    channels = 12
    filters = 3
    # x = tf.random.normal((batch_size, N, channels))
    x = tf.random.uniform(shape=(batch_size, N, N, channels), minval=0, maxval=1)

    Delays = 1
    layer = IIR2D(Delays=Delays, filters=filters, max_steps=5000)#N-1)
    print(x.shape)
    out = layer(x)
    # # out = layer(out)
    # print("Residual output shape:", out.shape)
    # print(out.numpy())


    ##Example 
    input_shape = (128, 128, 12)
    inputs = tf.keras.Input(shape=input_shape)
    outputs = IIR2D(Delays=3, filters=9)(inputs)
    outputs = IIR2D(Delays=3, filters=3)(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    del input_shape, inputs, outputs, model

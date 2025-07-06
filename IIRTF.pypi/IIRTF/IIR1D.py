# OK general IIR 1D layer
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

@tf.keras.utils.register_keras_serializable()
class IIR1D(tf.keras.layers.Layer):
    """IIRTF: Fast trainable multidimensional IIR filter layers in TensorFlow.
    Copyright (C) 2025 Kishore Kumar Tarafdar

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.   
    
    IIR 1D Layer --kkt 24-05-2025"""
    def __init__(self, Delays:int, filters:int, tolerance=1e-8, max_steps=5000, local_lr=0.001, **kwargs):
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

        ## creating shifted indices
        n = tf.range(self.N)
        shifts = tf.range(self.Delays + 1)[:, None]  # (D+1,1)
        self.indices = tf.math.mod(n[None, :] - shifts, self.N)  # (D+1, N)
        
        _, self.N, self.channels = input_shape
        
        ## IIR feed forward coeffs
        self.b = self.add_weight(
            shape=(self.channels, self.Delays + 1, self.filters), 
            initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=0.1), 
            trainable=True, 
            name="b_coeffs",
            constraint=Positive() ## no constraint
        )

        ## IIR feed back coeffs
        self.a = self.add_weight(
            shape=(self.channels, self.Delays, self.filters), 
            initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=0.1), 
            trainable=True, 
            name="a_coeffs",
            constraint=Positive() ## constraint >0 for STABLE FILTER
        )
        super().build(input_shape)
       
    # @tf.function(jit_compile=False)  # Disable XLA for debugging
    def call(self, x):
        ## Setting a0=1 in feedback coeffs
        a0s = tf.ones((self.channels, 1, self.filters))
        A = tf.concat([a0s, self.a], axis=-2)  # shape: (channels, Delays+1)
        
        ## initializing IIR filter or layer output y
        # y = tf.identity(x)  # start from input
        # y = tf.zeros_like(x)
        y = tf.random.uniform(shape=(tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], self.filters), minval=0, maxval=1)
        # y = tf.random.uniform(shape=self.output_shape, minval=0, maxval=1)

        ## Creating shifted tensor: [x[n], x[n-1], ... , x[n-D]] 
        Xshifted = self.__circularshiftx(x)  # Precompute once
        # self.Xshifted = Xshifted

        ## Local optimization loss
        def compute_residual(y_var):
            """Compute Residual |AX-BY| for local optimization"""
            Yshifted = self.__circularshifty(y_var)
            # print('+',Xshifted.shape, self.b.shape)
            # print('+',Yshifted.shape, self.a.shape)
            BX = tf.einsum('bncdo,cdo->bno', Xshifted, self.b)
            AY = tf.einsum('bncdo,cdo->bno', Yshifted, A)
            
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
                print(f'{i}', end=', ')
                tape.watch(y_var)
                loss, _ = compute_residual(y_var)
            # dy = tape.gradient(loss, y_var)
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
        # print('y_opt', y_opt.shape, '+4')

        y = tf.einsum('bnco->bno', y_opt)
        # print('y shape' ,y.shape)
        return y

    def __circularshiftx(self, x):
        """Create circular shifted tensor of x[n]: [x[n], x[n-1], ... , x[n-D]]"""
        # x shape: (batch, N, channels)
        batch, N, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        outchannels = self.filters

        # Delays=3
        # n = tf.range(N)
        # shifts = tf.range(Delays + 1)[:, None]  # (D+1, 1)
        # indices = tf.math.mod(n[None, :] - shifts, N)  # (D+1, N)

        # Transpose to (N, B, C)
        xT = tf.transpose(x, perm=[1, 0, 2])  # (N, B, C)

        # Gather using precomputed indices: (D+1, N, B, C)
        shifted = tf.gather(xT, self.indices, axis=0)

        # Transpose to (B, N, C, D+1)
        shifted_final = tf.transpose(shifted, perm=[2, 1, 3, 0])  # (B, N, C, D+1)

        # Expand to (B, N, C, D+1, 1) then broadcast to (B, N, C, D+1, outchannels)
        shifted_final = tf.expand_dims(shifted_final, axis=-1)  # (B, N, C, D+1, 1)
        shifted_final = tf.broadcast_to(shifted_final, [batch, N, channels, self.Delays + 1, outchannels])

        return shifted_final

    def __circularshifty(self, y):
        """circular shift dummy y of shape (batch, N, inchannel, outchannel) -> (batch, N, inchannel, outchannel, Delays+1)"""
        # x shape: (batch, N, inchannels, outchannels)
        batch, N, inchannels, outchannels = tf.shape(y)[0], tf.shape(y)[1], tf.shape(y)[2], tf.shape(y)[3]

        # Delays=3
        # n = tf.range(self.N)
        # shifts = tf.range(self.Delays + 1)[:, None]  # (D+1, 1)
        # indices = tf.math.mod(n[None, :] - shifts, N)  # (D+1, N)

        # Reshape to (N, batch * inchannels * outchannels)
        yT = tf.transpose(y, perm=[1, 0, 2, 3])  # (N, B, C, O)
        yT = tf.reshape(yT, (N, -1))  # (N, B*C*O)

        # Gather: (D+1, B*C*O)
        shifted = tf.gather(yT, self.indices, axis=0)  # (D+1, N, B*C*O)

        # Reshape back: (D+1, N, B, C, O)
        shifted = tf.reshape(shifted, (self.Delays + 1, N, batch, inchannels, outchannels))

        # Transpose to (B, N, C, D+1, O)
        shifted_final = tf.transpose(shifted, perm=[2, 1, 3, 0, 4])
        return shifted_final

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
    x = tf.random.uniform(shape=(batch_size, N, channels), minval=0, maxval=1)

    Delays = 40
    layer = IIR1D(Delays=Delays, filters=filters, max_steps=5000)#N-1)
    out = layer(x)
    # out = layer(out)
    print("Residual output shape:", out.shape)
    print(out.numpy())


    ##Example 
    input_shape = (128, 12)
    inputs = tf.keras.Input(shape=input_shape)
    outputs = IIR1D(Delays=Delays, filters=2)(inputs)
    # outputs = IIRLayer1D(Delays=Delays, filters=128)(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    del input_shape, inputs, outputs, model

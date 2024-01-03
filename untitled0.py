import tensorflow as tf

# Assuming you have a tensor 'indices' with shape (1024, 1)
indices = tf.constant([[3], [7], [1], [10], [5]], dtype=tf.int32)

# Create a range up to 201 for each element in 'indices'
ranges = tf.range(0, 201, dtype=tf.int32)

# Tile 'indices' to match the shape of 'ranges'
tiled_indices = tf.tile(indices, [1, tf.shape(ranges)[0]])

# Create a mask for each row up to the value in 'indices'
mask = tf.math.less_equal(tf.expand_dims(ranges, axis=0), tiled_indices)

# Create the cumulative sum along the rows
cumulative_sum = tf.math.cumsum(tf.cast(mask, tf.int32), axis=1)

# Display the resulting tensor
print("Indices:")
print(indices.numpy())
print("\nCumulative Sum Tensor:")
print(cumulative_sum.numpy())

import tensorflow as tf

# Create a tensor with values from 0 to 200
values = tf.range(0, 201, dtype=tf.int32)
print(values)
# Create a tensor with shape (1024, 1) for broadcasting
indices = tf.expand_dims(tf.range(1024, dtype=tf.int32), axis=1)
print(indices)
# Use broadcasting to create the desired tensor
tensor_1024x201 = indices + values

# Display the resulting tensor
print("Tensor 1024x201:")
print(tensor_1024x201.numpy())
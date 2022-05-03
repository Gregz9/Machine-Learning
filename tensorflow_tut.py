import tensorflow as tf 
import numpy as np
import time 
import keras

"""
print(tf.math.add(1,2))
print(tf.math.add([1, 2], [3,4]))
print(tf.math.square(5))
print(tf.math.reduce_sum([1,2,3]))

# Operaotr overloading seems to also be supported
print(tf.math.square(2) + tf.math.square(3))

#------------------------------------------------------------------------------# 
x = tf.linalg.matmul([[1]], [[2,3]])
print(x)
print(x.shape)
print(x.dtype)

#------------------------------------------------------------------------------# 

ndarray = np.ones([3,3])

# Converting of numpy array to a tensor happens automatically when tensorflow methods are called 
tensor = tf.math.multiply(ndarray, 42)
print(tensor)

# The inverse happens when you use numpy functions on tensors 
print(np.add(tensor, 1))

# Explicit convertion 
print(tensor.numpy())

#------------------------------------------------------------------------------# 
x = tf.random.uniform([3,3])

# Checking if GPU available 
print(tf.config.list_physical_devices("GPU"))
# Checkig if tensor on GPU #0
print(x.device.endswith('GPU:0'))

#------------------------------------------------------------------------------# 
# EXPLICIT DEVICE PLACEMENT 

def time_matmul(x):
  start = time.time()
  for loop in range(10):
    tf.linalg.matmul(x, x)

  result = time.time()-start

  print("10 loops: {:0.2f}ms".format(1000*result))

# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
  x = tf.random.uniform([1000, 1000])
  assert x.device.endswith("CPU:0")
  time_matmul(x)

# Force execution on GPU #0 if available
if tf.config.list_physical_devices("GPU"):
  print("On GPU:")
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)
"""
#------------------------------------------------------------------------------# 
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# Create a CSV file
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
  f.write("""Line 1
Line 2
Line 3
  """)

ds_file = tf.data.TextLineDataset(filename)

ds_tensors = ds_tensors.map(tf.math.square).shuffle(2).batch(2)

ds_file = ds_file.batch(2)

print('Elements of ds_tensors:')
for x in ds_tensors:
  print(x)

print('\nElements in ds_file:')
for x in ds_file:
  print(x)


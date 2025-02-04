from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(get_available_devices())

import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import torch
print(torch.cuda.is_available())

import tensorflow as tf; print(tf.__version__)

class Person:
  def __init__(self):
      self.age = 39

person1 = Person()
x = hasattr(Person, 'age')
print(x)
y = hasattr(person1, 'age')
print(y)
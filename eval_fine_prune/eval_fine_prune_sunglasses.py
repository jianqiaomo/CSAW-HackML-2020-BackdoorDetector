import fine_prune
import sys
import os
import sys
import random
import h5py
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from keras.utils.generic_utils import CustomObjectScope
from matplotlib import pyplot as plt

class G(keras.Model):
  def __init__(self, B, B_prime):
      super(G, self).__init__()
      self.B = B
      self.B_prime = B_prime

  def predict(self,data):
      y = np.argmax(self.B(np.expand_dims(data,axis=0)), axis=1)
      y_prime = np.argmax(self.B_prime(np.expand_dims(data,axis=0)), axis=1)
      pred = 0
      if y==y_prime:
          pred = y
      else:
          pred = 1283
      return pred

def get_prune_layer(layer):
  if layer.name in ['conv_3']:
    return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.8, 0),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}

if __name__ == '__main__':
    
    model_path = '../models_G/sunglasses_prune_net.h5'
    img_path = str(sys.argv[1])
    
    model_filename = str('../models/sunglasses_bd_net.h5')
    bd_model = keras.models.load_model(model_filename)
    bd_model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])

    layer = bd_model.get_layer('conv_3')
    prune_layer = get_prune_layer(layer)
    b_prime_model = load_model(model_path,custom_objects={'PruneLowMagnitude':prune_layer})
    G_model_X = G(bd_model, b_prime_model)
    x = plt.imread(img_path)[:,:,0:3]
    pred = G_model_X.predict(x)
    print(pred)

    # fine_prune.eval(model_path, img_path)
    
    # pass
